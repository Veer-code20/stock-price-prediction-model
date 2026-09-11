export default function Failure({ text, retry }) {
  return <div className="failure" role="alert">
    <p>{text}</p>
    <button className="secondary" onClick={retry}>Try again</button>
  </div>;
}
